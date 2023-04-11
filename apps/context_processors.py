from django.conf import settings
from apps.home.views import navig_profilePicture


def cfg_assets_root(request):

    return {'ASSETS_ROOT': settings.ASSETS_ROOT}


def profile_picture(request):
    if request.user.is_authenticated and hasattr(request.user, 'userprofile'):
        gender=request.user.userprofile.gender
        encoded_data = navig_profilePicture(request)['encodedData']
    else:
        encoded_data = {}
        gender={}
    return ({'encoded_data': encoded_data,'gender':gender})
