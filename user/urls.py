from django.urls import path
from . import views

urlpatterns = [
    path('signup/', views.signup, name='signup'),
    path('auth/login/', views.login, name='login'),
    path('auth/logout/', views.logout, name='logout'),
    path('auth/refresh/', views.refresh_token, name='refresh_token'),
    path('profile/', views.profile, name='user_profile'),
    path('update/', views.update_profile, name='update_profile'),
    path('change-password/', views.change_password, name='change_password'),
    path('delete/', views.delete_account, name='delete_account'),
    path('login-history/', views.login_history, name='login_history'),
    path('stats/', views.user_stats, name='user_stats'),
    path('get_user_id/', views.get_user_id, name='get_user_id'),
]