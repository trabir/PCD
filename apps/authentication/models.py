
from django.contrib.auth.models import User
from django.db import models

class UserProfile(models.Model):
    id=models.BigAutoField(primary_key=True)
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    avatar = models.BinaryField(null=True, editable = True)
    d_birth = models.DateField()
    gender = models.CharField(max_length=1)
    fb_url= models.URLField(max_length=50, null=True)
    twitter_url= models.URLField(max_length=50, null=True)
    linkedin_url= models.URLField(max_length=50, null=True)
