# -*- encoding: utf-8 -*-

from django.urls import path
from .views import login_view, register_user, logout_user

urlpatterns = [
    path('login/', login_view, name="login"),
    path('register/', register_user, name="register"),
    path("logout/", logout_user, name="logout")
]
