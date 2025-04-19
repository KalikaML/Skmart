# prediction_project/urls.py

from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    path('accounts/', include('allauth.urls')),  # For authentication
    path('', include('mainapp.urls')),  # Your app urls
]
