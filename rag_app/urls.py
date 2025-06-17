from django.urls import path
from . import views


urlpatterns = [
    path('chat', views.chat, name='chat'),
    path('create_embedding', views.create_embedding, name='create_embedding'),
]