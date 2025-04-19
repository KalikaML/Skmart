# mainapp/urls.py

from django.urls import path
from . import views

app_name = 'mainapp'

urlpatterns = [
    path('', views.dashboard, name='dashboard'),
    path('api/dashboard-data/', views.get_dashboard_data, name='dashboard_data'),
    path('api/real-time-data/', views.get_real_time_data, name='real_time_data'),
    path('api/predictions/', views.get_predictions, name='predictions'),

    path('chatbot/', views.chatbot_view, name='chatbot'),
    path('chatbot/api/message/', views.chatbot_message, name='chatbot_message'),
    path('chatbot/api/history/', views.get_chat_history, name='chat_history'),
    path('chatbot/api/new-session/', views.new_chat_session, name='new_chat_session'),
]
