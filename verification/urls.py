# verification/urls.py
from django.urls import path
from . import views

urlpatterns = [
    path('', views.upload_signature, name='upload_signature'),
]
