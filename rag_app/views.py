from django.http import JsonResponse
import logging
from typing import List, Dict, Any
from rest_framework.permissions import IsAuthenticated
from rest_framework.decorators import api_view, permission_classes
from qdrant_client import QdrantClient
from langchain.chat_models import init_chat_model
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from qdrant_client.models import PointStruct, VectorParams, Distance
import os
from rest_framework import status
import json
from urllib.parse import urlparse
import requests
from dotenv import load_dotenv
from django.utils import timezone
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.document_loaders import RecursiveUrlLoader
from bs4 import BeautifulSoup
import uuid
from datetime import datetime

logger = logging.getLogger(__name__)
load_dotenv()

try:
    client = QdrantClient(url=os.getenv("QDRANT_URL"), port=6333, api_key=os.getenv("QDRANT_API_KEY"))
    logger.info("Qdrant client initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize Qdrant client: {str(e)}")
    client = None
# "/etc/secrets/sahayakai-462506-9ab8250eff98.json" for production
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/etc/secrets/sahayakai-462506-9ab8250eff98.json"

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
        if parsed.hostname in ['localhost', '127.0.0.1'] or (parsed.hostname and parsed.hostname.startswith('192.168.')):
            return False

        # Test if URL is accessible
        response = requests.head(url, timeout=10, allow_redirects=True)
        return response.status_code == 200
    except Exception:
        return False



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
- Analyze the provided {context} to understand relevant information or code
- Provide clear explanations based ONLY on the provided {context}
- Include code examples when available in the {context}
- Reference the non clickable source URL for each piece of information you use, give the full url only once.
- while referencing the source URL, include the #title of the document in the url, e.g. https://example.com/document/#<topic_title>
- If multiple chunks are relevant, integrate them coherently
- If the {context} doesn't contain enough information, say so honestly
- For navigation questions, provide specific URLs or section references when available

Format your responses using simple text for clarity. Always cite your sources.

Important: Only use information from the provided {context}. Do not make assumptions or add information not present in the {context}."""


def retrieve_relevant_context(query: str, limit: int = 5) -> List[Dict[str, Any]]:
    """Retrieve relevant context from user's vector database"""
    try:
        if not client:
            logger.error("Client model not initialized")
            return []
        elif not embeddings_model:
            logger.error("Embeddings model not initialized")
            return []
            
        print(f"Retrieving context for query: {query}")
        collection_name = 'ChaiDocs'

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
                if point.payload:
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
                {'success': False, 'code': 400 , 'error': "Message is required"},
                status=status.HTTP_400_BAD_REQUEST
            )

        # Retrieve relevant context from user's documents
        contexts = retrieve_relevant_context(user_message, limit=10)

        if not contexts:
            return JsonResponse({
                'success': True,
                'code': 404,
                "response": "I don't have any relevant documentation to answer your question. Please upload some documentation first using the /create_embedding endpoint.",
                "user_message": user_message,
                "sources": [0],
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
            'success': True,
            'code': 200,
            "response": ai_response,
            "user_message": user_message,
            "sources": sources,
            "context_chunks_used": len(contexts),
            "timestamp": timezone.now().isoformat(),
            "user_id": request.user.id
        }, status=status.HTTP_200_OK)

    except json.JSONDecodeError:
        return JsonResponse({
            'success': False,
            'code': 400,
            'error': "Invalid JSON format"
        },
            status=status.HTTP_400_BAD_REQUEST
        )
    except Exception as e:
        logger.error(f"Chat API error: {str(e)}")
        return JsonResponse({
            'success': False,
            'code': 500,
            'error': "An error occurred while processing your request"
                             },
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@api_view(['POST'])
@permission_classes([IsAuthenticated])
def create_embedding(request):
    """
    Create embeddings for user documents and store in Qdrant
    """
    try:
        if not client or not embeddings_model:
            return JsonResponse({
                'success': False,
                'code': 500,
                'error': "Vector database or embeddings model not initialized"
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        data = json.loads(request.body) if request.body else {}
        document_url = data.get('link')

        if not document_url or not validate_url(document_url):
            return JsonResponse({
                'success': False,
                'code': 400,
                'error': "Invalid or inaccessible document URL"
            }, status=status.HTTP_400_BAD_REQUEST)

        collection_name = request.collection_name
        
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=768, distance=Distance.COSINE)
        )
        print(f"Created collection: {collection_name}")

        loader = RecursiveUrlLoader(
            url=document_url,
            max_depth=1000,
            extractor=lambda html: BeautifulSoup(html, "html.parser").get_text()
        )

        docs = loader.load()
        
        if not docs:
            return JsonResponse({
                'success': False,
                'code': 400,
                'error': "No content could be extracted from the provided URL"
            }, status=status.HTTP_400_BAD_REQUEST)

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=400)
        
        # Process documents and create points
        points = []
        total_docs = len(docs)
        
        for doc_index, doc in enumerate(docs, 1):
            logger.info(f"Processing document {doc_index}/{total_docs}")
            
            # Split text into chunks
            chunks = text_splitter.create_documents(
                texts=[doc.page_content],
                metadatas=[{
                    "source_url": doc.metadata.get('source', document_url),
                    "title": doc.metadata.get('title', f"Document {doc_index}"),
                    "timestamp": datetime.now().isoformat(),
                    "chunk_size": 1000,
                    "chunk_overlap": 400,
                    "content_type": "documentation",
                    "original_url": document_url
                }]
            )
            
            chunk_embeddings = embeddings_model.embed_documents([chunk.page_content for chunk in chunks])
            
            for chunk_index, (chunk, embedding) in enumerate(zip(chunks, chunk_embeddings)):
                point = PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embedding,
                    payload={
                        "page_content": chunk.page_content,
                        "metadata": chunk.metadata,
                        "doc_index": doc_index,
                        "chunk_index": chunk_index,
                        "total_chunks": len(chunks),
                        "processing_timestamp": datetime.now().isoformat()
                    }
                )
                points.append(point)

        batch_size = 100
        total_batches = (len(points) + batch_size - 1) // batch_size
        
        logger.info(f"Inserting {len(points)} vectors in {total_batches} batches...")
        
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            current_batch = i // batch_size + 1
            try:
                client.upsert(
                    collection_name=collection_name,
                    points=batch
                )
                logger.info(f"Inserted batch {current_batch}/{total_batches}")
            except Exception as e:
                logger.error(f"Error inserting batch {current_batch}: {str(e)}")
                return JsonResponse({
                    'success': False,
                    'code': 500,
                    'error': f"Failed to insert batch {current_batch}: {str(e)}"
                }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        logger.info(f"Successfully created and stored {len(points)} vectors in collection: {collection_name}")

        return JsonResponse({
            'success': True,
            'code': 201,
            "message": f"Documents embedded and stored successfully ({len(points)} chunks processed from {total_docs} documents)",
            "collection_name": collection_name,
            "documents_processed": total_docs,
            "chunks_created": len(points),
            "user_id": request.user.id
        }, status=status.HTTP_201_CREATED)

    except json.JSONDecodeError:
        return JsonResponse({
            'success': False,
            'code': 400,
            'error': "Invalid JSON format"
        }, status=status.HTTP_400_BAD_REQUEST)
    except Exception as e:
        logger.error(f"Embedding API error: {str(e)}")
        return JsonResponse({
            'success': False,
            'code': 500,
            'error': "An error occurred while processing your request"
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)