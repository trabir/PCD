from django.urls import path, re_path
from apps.home import views
from .views import update_selected_secteur, onChangeHistoricalData, delete_avatar

urlpatterns = [

    # The home page 
    path('', views.index, name='home'),
    path('update-selected-secteur/', update_selected_secteur, name='update-selected-secteur'),
    path('update-selected-societe/', onChangeHistoricalData, name='update-selected-societe'),
    
    path('user.html', views.user, name='user'),
    path('delete-avatar/', delete_avatar, name='delete-avatar'),
    
    # Matches any html file
    re_path(r'^.*\.*', views.pages, name='pages'),

]
