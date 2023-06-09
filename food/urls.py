from django.urls import path
from . import views
from users.views import register

urlpatterns = [
    path('', views.home, name="food-home"),
    path('about/', views.about, name="food-about"),
    path('register/', register, name='register'),
    path('partners', views.partners, name='partners')
]
