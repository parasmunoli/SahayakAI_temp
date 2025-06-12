from django.http import JsonResponse
import logging
from typing import List, Dict, Any
import tempfile
from rest_framework.permissions import IsAuthenticated
from rest_framework.decorators import api_view, permission_classes
from qdrant_client import QdrantClient
from langchain.chat_models import init_chat_model
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from qdrant_client.models import Distance, VectorParams
import os
from rest_framework import status
import json
from urllib.parse import urlparse
import requests
from dotenv import load_dotenv
from django.utils import timezone

logger = logging.getLogger(__name__)
load_dotenv()

# Initialize Qdrant client
try:
    client = QdrantClient(url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_API_KEY"))
    logger.info("Qdrant client initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize Qdrant client: {str(e)}")
    client = None
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/etc/secrets/sahayakai-462506-9ab8250eff98.json"
# Initialize embeddings model
try:
    embeddings_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    logger.info("Embeddings model initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize embeddings model: {str(e)}")
    embeddings_model = None


def validate_url(url: str) -> bool:
    """Validate if URL is accessible and safe"""
    try:
        parsed = urlparse(url)
        if parsed.scheme not in ['http', 'https']:
            return False

        # Basic security check - avoid local/private IPs
        if parsed.hostname in ['localhost', '127.0.0.1'] or parsed.hostname.startswith('192.168.'):
            return False

        # Test if URL is accessible
        response = requests.head(url, timeout=10, allow_redirects=True)
        return response.status_code == 200
    except Exception:
        return False


def get_user_collection_name(user_id: int) -> str:
    """Generate collection name for user"""
    return f"user_{user_id}_docs"


def ensure_collection_exists(collection_name: str) -> bool:
    """Ensure Qdrant collection exists"""
    try:
        if not client:
            return False

        collections = client.get_collections().collections
        collection_names = [col.name for col in collections]

        if collection_name not in collection_names:
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=768, distance=Distance.COSINE)
            )
            logger.info(f"Created collection: {collection_name}")

        return True
    except Exception as e:
        logger.error(f"Error ensuring collection exists: {str(e)}")
        return False



def generate_system_prompt(context) -> str:
    """Generate system prompt for the RAG chat assistant"""
    return f"""You are a technical documentation assistant. You help developers understand codebases and documentation by analyzing provided {context} chunks.

Your responsibilities:
- Analyze the provided {context} to understand relevant code/documentation
- Provide clear, accurate explanations based ONLY on the provided {context}
- Include code examples when available in the {context}
- Reference the source URL for each piece of information you use
- If multiple chunks are relevant, integrate them coherently
- If the {context} doesn't contain enough information, say so honestly
- For navigation questions, provide specific URLs or section references when available

Format your responses using Markdown for clarity. Always cite your sources.

Important: Only use information from the provided {context}. Do not make assumptions or add information not present in the {context}."""


def retrieve_relevant_context(query: str, limit: int = 5) -> List[Dict[str, Any]]:
    """Retrieve relevant context from user's vector database"""
    try:
        print(f"Retrieving context for query: {query}")
        collection_name = 'ChaiDocs'  # Default collection name for documentation

        # Generate query embedding
        query_embedding = embeddings_model.embed_query(query)
        print(f"Query embedding: {query_embedding}")

        # Search for similar vectors using query_points
        search_results = client.query_points(
            collection_name=collection_name,
            query=query_embedding,
            limit=limit,
        )

        print(f"Search results: {search_results}")

        contexts = []
        for point in search_results.points:  # Access the points attribute
            try:
                contexts.append({
                    "content": point.payload.get('page_content'),
                    "source_url": point.payload.get('metadata', {}).get('source_url'),
                    "title": point.payload.get('metadata', {}).get('title'),
                    "score": point.score if hasattr(point, 'score') else None
                })
            except Exception as e:
                logger.error(f"Error processing search result: {str(e)}")
                continue

        return contexts

    except Exception as e:
        logger.error(f"Error retrieving context: {str(e)}")
        return []


@api_view(['POST'])
@permission_classes([IsAuthenticated])
def chat(request):
    """
    Chat endpoint with RAG functionality - retrieves relevant context and generates response
    """
    try:
        # Parse request data
        data = json.loads(request.body) if request.body else {}
        user_message = data.get('message')

        if not user_message:
            return JsonResponse(
                {"error": "Message is required"},
                status=status.HTTP_400_BAD_REQUEST
            )

        # Retrieve relevant context from user's documents
        contexts = retrieve_relevant_context(user_message, limit=10)

        if not contexts:
            return JsonResponse({
                "success": True,
                "response": "I don't have any relevant documentation to answer your question. Please upload some documentation first using the /create_embedding endpoint.",
                "user_message": user_message,
                "sources": [],
                "user_id": request.user.id
            })

        # Prepare context for LLM
        context_text = "\n\n---\n\n".join([
            f"**Source:** {ctx['source_url']}\n**Title:** {ctx['title']}\n**Content:** {ctx['content']}"
            for ctx in contexts
        ])

        # Initialize LLM
        llm = init_chat_model(model_provider='google_genai', model="gemini-2.0-flash")

        # Prepare messages with system prompt and context
        messages = [
            {
                'role': 'system',
                'content': generate_system_prompt(context_text)
            },
            {
                'role': 'user',
                'content': user_message
            }
        ]

        # Generate response
        response = llm.invoke(messages)
        ai_response = response.content if hasattr(response, 'content') else str(response)

        # Prepare sources for frontend
        sources = [
            {
                "url": ctx['source_url'],
                "title": ctx['title'],
                "relevance_score": round(ctx['score'], 3)
            }
            for ctx in contexts
        ]

        logger.info(f"RAG chat request from user {request.user.id}: {user_message[:100]}...")

        return JsonResponse({
            "success": True,
            "response": ai_response,
            "user_message": user_message,
            "sources": sources,
            "context_chunks_used": len(contexts),
            "timestamp": timezone.now().isoformat(),
            "user_id": request.user.id
        }, status=status.HTTP_200_OK)

    except json.JSONDecodeError:
        return JsonResponse(
            {"error": "Invalid JSON format"},
            status=status.HTTP_400_BAD_REQUEST
        )
    except Exception as e:
        logger.error(f"Chat API error: {str(e)}")
        return JsonResponse(
            {"error": "An error occurred while processing your request"},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )