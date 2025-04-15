from django.contrib import admin
from django.urls import path, include
from . import views
# from .views import page_view

urlpatterns = [
    path('', views.chatbot_views, name='chatbot_views'),
]