from django.db import models
from django.contrib.auth.models import User


class Post(models.Model):
    image = models.ImageField(upload_to='media/fruit_images/')
    prediction = models.CharField(max_length=200, blank=True)
    ripeness = models.DecimalField(max_digits=3, decimal_places=2, default=0.0)
    date_posted = models.DateTimeField(auto_now_add=True)
    author = models.ForeignKey(User, on_delete=models.CASCADE)
