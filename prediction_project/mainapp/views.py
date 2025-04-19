# mainapp/views.py

from django.shortcuts import render, get_object_or_404
from django.contrib.auth.decorators import login_required
from rest_framework.decorators import api_view
from rest_framework.response import Response
from .models import HistoricalData, Prediction, ChatSession, ChatMessage
from django.utils import timezone
import numpy as np
import pandas as pd
import datetime
import google.generativeai as genai
import os

# Configure Gemini API
genai.configure(api_key=os.getenv('GEMINI_API_KEY'))

@login_required
def dashboard(request):
    return render(request, 'mainapp/dashboard.html')

@api_view(['GET'])
@login_required
def get_dashboard_data(request):
    # Get last 100 historical data points
    historical = HistoricalData.objects.order_by('-timestamp')[:100]
    historical = reversed(historical)  # oldest first

    # Get last 100 predictions for user
    predictions = Prediction.objects.filter(user=request.user).order_by('-timestamp')[:100]
    predictions = reversed(predictions)

    hist_data = [{
        'timestamp': hd.timestamp.isoformat(),
        'value': hd.value,
        'category': hd.category
    } for hd in historical]

    pred_data = [{
        'timestamp': p.timestamp.isoformat(),
        'value': p.predicted_value,
        'category': p.category,
        'confidence': p.confidence
    } for p in predictions]

    return Response({
        'historical_data': hist_data,
        'predictions': pred_data,
    })

@api_view(['GET'])
@login_required
def get_real_time_data(request):
    # Simulate real-time data by adding noise to last historical data
    categories = HistoricalData.objects.values_list('category', flat=True).distinct()
    now = timezone.now()
    real_time_data = []

    for cat in categories:
        last = HistoricalData.objects.filter(category=cat).order_by('-timestamp').first()
        if last:
            noise = np.random.normal(0, last.value * 0.05)
            new_value = last.value + noise
            # Save new real-time data point
            new_point = HistoricalData.objects.create(timestamp=now, value=new_value, category=cat)
            real_time_data.append({
                'timestamp': new_point.timestamp.isoformat(),
                'value': new_point.value,
                'category': new_point.category,
            })

    return Response({'real_time_data': real_time_data})

@api_view(['GET'])
@login_required
def get_predictions(request):
    # Simple linear regression prediction example (can be replaced with your model)
    historical = HistoricalData.objects.order_by('timestamp')
    df = pd.DataFrame(list(historical.values()))
    if df.empty:
        return Response({'status': 'error', 'message': 'No historical data'})

    predictions = []
    for category in df['category'].unique():
        cat_df = df[df['category'] == category]
        if len(cat_df) < 10:
            continue
        # Use index as X, value as y
        X = np.arange(len(cat_df)).reshape(-1, 1)
        y = cat_df['value'].values
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        model.fit(X, y)
        next_x = np.array([[len(cat_df)]])
        pred_val = model.predict(next_x)[0]
        # Save prediction
        pred_obj = Prediction.objects.create(user=request.user, predicted_value=pred_val, category=category, confidence=0.8)
        predictions.append({
            'timestamp': pred_obj.timestamp.isoformat(),
            'value': pred_obj.predicted_value,
            'category': pred_obj.category,
            'confidence': pred_obj.confidence,
        })

    return Response({'status': 'success', 'predictions': predictions})

@login_required
def chatbot_view(request):
    return render(request, 'mainapp/chatbot.html')

@api_view(['POST'])
@login_required
def chatbot_message(request):
    message = request.data.get('message')
    session_id = request.data.get('session_id')

    if session_id:
        session = get_object_or_404(ChatSession, id=session_id, user=request.user)
    else:
        session = ChatSession.objects.create(user=request.user)

    # Save user message
    ChatMessage.objects.create(session=session, role='user', content=message)

    # Prepare conversation history for context
    history = ChatMessage.objects.filter(session=session).order_by('timestamp')
    conversation = [{"role": msg.role, "content": msg.content} for msg in history]

    # Generate Gemini response
    try:
        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content(message)
        response_text = response.text

        # Save assistant message
        ChatMessage.objects.create(session=session, role='assistant', content=response_text)

        return Response({'status': 'success', 'session_id': session.id, 'message': response_text})
    except Exception as e:
        return Response({'status': 'error', 'message': str(e)})

@api_view(['GET'])
@login_required
def get_chat_history(request, session_id=None):
    if session_id:
        session = get_object_or_404(ChatSession, id=session_id, user=request.user)
    else:
        session = ChatSession.objects.filter(user=request.user).order_by('-started_at').first()
        if not session:
            session = ChatSession.objects.create(user=request.user)

    messages = ChatMessage.objects.filter(session=session).order_by('timestamp')
    message_list = [{'role': m.role, 'content': m.content, 'timestamp': m.timestamp.isoformat()} for m in messages]

    return Response({'status': 'success', 'session_id': session.id, 'messages': message_list})

@api_view(['POST'])
@login_required
def new_chat_session(request):
    session = ChatSession.objects.create(user=request.user)
    return Response({'status': 'success', 'session_id': session.id})

