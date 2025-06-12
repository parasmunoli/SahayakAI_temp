from django.db import models
from django.conf import settings

class Document(models.Model):
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    url = models.URLField()
    title = models.CharField(max_length=500)
    status = models.CharField(max_length=50)  # processing, completed, failed
    created_at = models.DateTimeField(auto_now_add=True)


class ChatSession(models.Model):
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    question = models.TextField()
    response = models.TextField()
    sources_used = models.JSONField()
    created_at = models.DateTimeField(auto_now_add=True)