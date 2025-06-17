"""
URL configuration for SahayakAI project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.urls import include, path
from django.http import HttpResponse, JsonResponse


def home_view(request):
    return HttpResponse("Welcome to SahayakAI")

def status_view(request):
    return JsonResponse({
        "success": True,
        "code": 200,
        "status": "ok",
        "message": "SahayakAI is running."
    })


urlpatterns = [
    path('', home_view),
    path('status', status_view),
    path('api/user/', include('user.urls')),
    path('api/rag-app/', include('rag_app.urls')),
]
