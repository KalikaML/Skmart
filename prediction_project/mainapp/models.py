from django.db import models
from django.contrib.auth.models import User

class HistoricalData(models.Model):
    timestamp = models.DateTimeField()
    value = models.FloatField()
    category = models.CharField(max_length=100)

    class Meta:
        ordering = ['timestamp']

    def __str__(self):
        return f"{self.category} at {self.timestamp}"

class Prediction(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='predictions')
    timestamp = models.DateTimeField(auto_now_add=True)
    predicted_value = models.FloatField()
    category = models.CharField(max_length=100)
    confidence = models.FloatField(default=0.0)

    class Meta:
        ordering = ['-timestamp']

    def __str__(self):
        return f"Prediction for {self.category} by {self.user.username} at {self.timestamp}"

class ChatSession(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='chat_sessions')
    started_at = models.DateTimeField(auto_now_add=True)
    is_active = models.BooleanField(default=True)

    def __str__(self):
        return f"Session {self.id} for {self.user.username}"

class ChatMessage(models.Model):
    ROLE_CHOICES = [('user', 'User'), ('assistant', 'Assistant')]
    session = models.ForeignKey(ChatSession, on_delete=models.CASCADE, related_name='messages')
    role = models.CharField(max_length=10, choices=ROLE_CHOICES)
    content = models.TextField()
    timestamp = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['timestamp']

    def __str__(self):
        return f"{self.role} message in session {self.session.id}"

