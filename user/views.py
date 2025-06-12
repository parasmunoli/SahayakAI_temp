from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated, AllowAny
from rest_framework.response import Response
from django.contrib.auth import authenticate
import os
from rest_framework import status
from rest_framework_simplejwt.tokens import RefreshToken
from rest_framework_simplejwt.exceptions import TokenError
from django.contrib.auth.password_validation import validate_password
from django.core.exceptions import ValidationError
from .models import User, UserLoginHistory
from django.utils import timezone
import logging
from django.core.validators import validate_email
from dotenv import load_dotenv
from django.contrib.auth.models import AnonymousUser

load_dotenv()
logger = logging.getLogger(__name__)



def get_client_ip(request):
    """Extract client IP address from request"""
    x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
    if x_forwarded_for:
        ip = x_forwarded_for.split(',')[0]
    else:
        ip = request.META.get('REMOTE_ADDR')
    return ip


def get_user_agent(request):
    """Extract user agent from request"""
    return request.META.get('HTTP_USER_AGENT', '')


def get_tokens_for_user(user):
    """Generate JWT tokens for a user"""
    refresh = RefreshToken.for_user(user)

    # Add minimal custom claims (avoid sensitive data in JWT)
    refresh['username'] = user.email
    refresh['full_name'] = user.full_name
    # Keep user_id for JWT functionality - don't delete it

    return {
        'refresh': str(refresh),
        'access': str(refresh.access_token),
    }


def log_login_attempt(user, request, success=True):
    """Log login attempt with request metadata"""
    try:
        UserLoginHistory.objects.create(
            user=user,
            success=success,
            ip_address=get_client_ip(request),
            user_agent=get_user_agent(request),
            # location can be added later with GeoIP
        )
    except Exception as e:
        logger.error(f"Failed to log login attempt: {str(e)}")


@api_view(['POST'])
@permission_classes([AllowAny])
def signup(request):
    """User registration endpoint"""
    try:
        email = request.data.get('email', '').strip().lower()
        password = request.data.get('password', '')
        confirm_password = request.data.get('confirm_password', '')
        first_name = request.data.get('first_name', '').strip()
        last_name = request.data.get('last_name', '').strip()

        # Enhanced validation
        if not email or not password:
            return Response({
                'error': 'Email and password are required.'
            }, status=status.HTTP_400_BAD_REQUEST)

        # Validate email format
        try:
            validate_email(email)
        except ValidationError:
            return Response({
                'error': 'Invalid email format.'
            }, status=status.HTTP_400_BAD_REQUEST)

        if User.objects.filter(email=email).exists():
            return Response({
                'error': 'Email already exists.'
            }, status=status.HTTP_400_BAD_REQUEST)

        if password != confirm_password:
            return Response({
                'error': 'Passwords do not match.'
            }, status=status.HTTP_400_BAD_REQUEST)

        # Password validation
        try:
            validate_password(password)
        except ValidationError as e:
            return Response({
                'error': 'Password validation failed.',
                'details': list(e.messages)
            }, status=status.HTTP_400_BAD_REQUEST)

        # Create user
        user = User.objects.create_user(
            email=email,
            password=password,
            first_name=first_name,
            last_name=last_name
        )

        logger.info(f"New user registered: {user.email}")

        return Response({
            'message': 'User created successfully.',
            'user': {
                'id': user.id,
                'email': user.email,
                'first_name': user.first_name,
                'last_name': user.last_name,
            }
        }, status=status.HTTP_201_CREATED)

    except Exception as e:
        logger.error(f"Signup error: {str(e)}")
        return Response({
            'error': 'Failed to create user.'
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(['POST'])
@permission_classes([AllowAny])
def login(request):
    """User login endpoint"""
    try:
        email = request.data.get('email', '').strip().lower()
        password = request.data.get('password', '')

        if not email or not password:
            return Response({
                'error': 'Email and password are required.'
            }, status=status.HTTP_400_BAD_REQUEST)

        # Authenticate using email
        user = authenticate(request, username=email, password=password)

        if user is not None:
            if not user.is_active:
                return Response({
                    'error': 'Account is deactivated.'
                }, status=status.HTTP_401_UNAUTHORIZED)

            # Update last login
            user.last_login = timezone.now()
            user.save(update_fields=['last_login'])

            # Log successful login
            log_login_attempt(user, request, success=True)

            tokens = get_tokens_for_user(user)

            return Response({
                'message': 'Login successful.',
                'user': {
                    'id': user.id,
                    'username': user.email,
                    'email': user.email,
                    'first_name': user.first_name,
                    'last_name': user.last_name,
                    'full_name': user.full_name
                },
                'tokens': tokens
            }, status=status.HTTP_200_OK)
        else:
            # Log failed login attempt
            try:
                failed_user = User.objects.get(email=email)
                log_login_attempt(failed_user, request, success=False)
            except User.DoesNotExist:
                pass

            return Response({
                'error': 'Invalid email or password.'
            }, status=status.HTTP_401_UNAUTHORIZED)

    except Exception as e:
        logger.error(f"Login error: {str(e)}")
        return Response({
            'error': 'Login failed.'
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(['POST'])
@permission_classes([IsAuthenticated])
def logout(request):
    """User logout endpoint - blacklist refresh token"""
    try:
        refresh_token = request.data.get('refresh_token') or request.data.get('refresh')

        if not refresh_token:
            return Response({
                'error': 'Refresh token is required.'
            }, status=status.HTTP_400_BAD_REQUEST)

        token = RefreshToken(refresh_token)
        token.blacklist()

        logger.info(f"User {request.user.email} logged out successfully")

        return Response({
            'message': 'Logged out successfully.'
        }, status=status.HTTP_200_OK)

    except TokenError as e:
        return Response({
            'error': 'Invalid or expired token.'
        }, status=status.HTTP_400_BAD_REQUEST)
    except Exception as e:
        logger.error(f"Logout error: {str(e)}")
        return Response({
            'error': 'Logout failed.'
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(['POST'])
@permission_classes([AllowAny])  # Should be AllowAny for refresh
def refresh_token(request):
    """Refresh access token using refresh token"""
    try:
        refresh_token = request.data.get('refresh')

        if not refresh_token:
            return Response({
                'error': 'Refresh token is required.'
            }, status=status.HTTP_400_BAD_REQUEST)

        refresh = RefreshToken(refresh_token)
        access_token = refresh.access_token

        return Response({
            'access': str(access_token)
        }, status=status.HTTP_200_OK)

    except TokenError:
        return Response({
            'error': 'Invalid or expired refresh token.'
        }, status=status.HTTP_401_UNAUTHORIZED)
    except Exception as e:
        logger.error(f"Token refresh error: {str(e)}")
        return Response({
            'error': 'Token refresh failed.'
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def profile(request):
    """Get user profile information"""
    user = request.user

    return Response({
        'id': user.id,
        'username': user.username,
        'email': user.email,
        'first_name': user.first_name,
        'last_name': user.last_name,
        'full_name': user.full_name,
        'is_active': user.is_active,
        'created_at': user.created_at,
        'updated_at': user.updated_at,
        'last_login': user.last_login
    }, status=status.HTTP_200_OK)


@api_view(['PUT'])
@permission_classes([IsAuthenticated])
def update_profile(request):
    """Update user profile information"""
    try:
        user = request.user

        # Update allowed fields
        first_name = request.data.get('first_name', user.first_name).strip()
        last_name = request.data.get('last_name', user.last_name).strip()
        email = request.data.get('email', user.email).strip().lower()

        # Validate email format
        if email != user.email:
            try:
                validate_email(email)
            except ValidationError:
                return Response({
                    'error': 'Invalid email format.'
                }, status=status.HTTP_400_BAD_REQUEST)

            # Check if email is already taken
            if User.objects.filter(email=email).exists():
                return Response({
                    'error': 'Email already exists.'
                }, status=status.HTTP_400_BAD_REQUEST)

        # Update user fields
        user.first_name = first_name
        user.last_name = last_name
        user.email = email
        user.save()

        return Response({
            'message': 'Profile updated successfully.',
            'user': {
                'id': user.id,
                'username': user.username,
                'email': user.email,
                'first_name': user.first_name,
                'last_name': user.last_name,
                'full_name': user.full_name
            }
        }, status=status.HTTP_200_OK)

    except Exception as e:
        logger.error(f"Profile update error: {str(e)}")
        return Response({
            'error': 'Failed to update profile.'
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(['POST'])
@permission_classes([IsAuthenticated])
def change_password(request):
    """Change user password"""
    try:
        user = request.user
        old_password = request.data.get('old_password')
        new_password = request.data.get('new_password')
        confirm_password = request.data.get('confirm_password')

        if not old_password or not new_password:
            return Response({
                'error': 'Both old and new passwords are required.'
            }, status=status.HTTP_400_BAD_REQUEST)

        if confirm_password and new_password != confirm_password:
            return Response({
                'error': 'New passwords do not match.'
            }, status=status.HTTP_400_BAD_REQUEST)

        if not user.check_password(old_password):
            return Response({
                'error': 'Old password is incorrect.'
            }, status=status.HTTP_400_BAD_REQUEST)

        # Validate new password
        validate_password(new_password, user)
        user.set_password(new_password)
        user.save()

        logger.info(f"Password changed for user: {user.email}")

        return Response({
            'message': 'Password changed successfully.'
        }, status=status.HTTP_200_OK)

    except ValidationError as e:
        return Response({
            'error': 'Password validation failed.',
            'details': list(e.messages)
        }, status=status.HTTP_400_BAD_REQUEST)
    except Exception as e:
        logger.error(f"Password change error: {str(e)}")
        return Response({
            'error': 'Failed to change password.'
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def login_history(request):
    """Get a user's login history"""
    try:
        user = request.user
        limit = min(int(request.GET.get('limit', 10)), 50)  # Max 50 records
        offset = int(request.GET.get('offset', 0))

        history = UserLoginHistory.objects.filter(
            user=user
        ).order_by('-login_time')[offset:offset + limit]

        history_data = []
        for entry in history:
            history_data.append({
                'id': entry.id,
                'login_time': entry.login_time,
                'ip_address': entry.ip_address,
                'user_agent': entry.user_agent,
                'location': entry.location,
                'success': entry.success
            })

        total_count = UserLoginHistory.objects.filter(user=user).count()

        return Response({
            'login_history': history_data,
            'total_count': total_count,
            'limit': limit,
            'offset': offset
        }, status=status.HTTP_200_OK)

    except Exception as e:
        logger.error(f"Login history error: {str(e)}")
        return Response({
            'error': 'Failed to retrieve login history.'
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(['DELETE'])
@permission_classes([IsAuthenticated])
def delete_account(request):
    """Delete user account (soft delete - deactivate)"""
    try:
        user = request.user
        password = request.data.get('password')

        if not password:
            return Response({
                'error': 'Password is required to delete account.'
            }, status=status.HTTP_400_BAD_REQUEST)

        if not user.check_password(password):
            return Response({
                'error': 'Incorrect password.'
            }, status=status.HTTP_400_BAD_REQUEST)

        # Soft delete - deactivate account
        user.is_active = False
        user.save(update_fields=['is_active'])

        logger.info(f"Account deactivated: {user.email}")

        return Response({
            'message': 'Account deactivated successfully.'
        }, status=status.HTTP_200_OK)

    except Exception as e:
        logger.error(f"Account deletion error: {str(e)}")
        return Response({
            'error': 'Failed to delete account.'
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def user_stats(request):
    """Get user statistics"""
    try:
        user = request.user

        total_logins = UserLoginHistory.objects.filter(user=user, success=True).count()
        failed_logins = UserLoginHistory.objects.filter(user=user, success=False).count()
        last_successful_login = UserLoginHistory.objects.filter(
            user=user, success=True
        ).order_by('-login_time').first()

        stats = {
            'user_id': user.id,
            'username': user.username,
            'email': user.email,
            'account_created': user.created_at,
            'last_updated': user.updated_at,
            'last_login': user.last_login,
            'is_active': user.is_active,
            'total_successful_logins': total_logins,
            'total_failed_logins': failed_logins,
            'last_successful_login': last_successful_login.login_time if last_successful_login else None
        }

        return Response(stats, status=status.HTTP_200_OK)

    except Exception as e:
        logger.error(f"User stats error: {str(e)}")
        return Response({
            'error': 'Failed to retrieve user statistics.'
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_user_id(request):
    """Get the user ID of the authenticated user"""
    user = request.user
    return Response({
        'user_id': user.id,
        'username': user.username,
        'email': user.email
    }, status=status.HTTP_200_OK)