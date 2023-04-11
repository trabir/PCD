from django import forms
from django.contrib.auth.forms import UserCreationForm, UserChangeForm
from django.contrib.auth.models import User
from apps.authentication.models import UserProfile


class LoginForm(forms.Form):
    username = forms.CharField(
        widget=forms.TextInput(
            attrs={
                "placeholder": "Username",
                "class": "form-control"
            }
        ))
    password = forms.CharField(
        widget=forms.PasswordInput(
            attrs={
                "placeholder": "Password",
                "class": "form-control"
            }
        ))


class SignUpForm(UserCreationForm):
    avatar = forms.ImageField(
        required=False,
        widget=forms.FileInput(
            attrs={'class': 'custom-file-input',
                   'id': 'avatar'}
        ))

    first_name = forms.CharField(
        widget=forms.TextInput(
            attrs={
                "placeholder": "Firstname",
                "class": "form-control",
                "autofocus": "true"
            }
        ))
    last_name = forms.CharField(
        widget=forms.TextInput(
            attrs={
                "placeholder": "Lastname",
                "class": "form-control"
            }
        ))

    d_birth = forms.DateField(
        widget=forms.DateInput(
            attrs={
                'type': 'date',
                "class": "form-control"
            }
        ))

    gender = forms.ChoiceField(
        widget=forms.RadioSelect(attrs={'class': 'custom-control-input'}),
        choices=[('M', 'Male'), ('F', 'Female')]
    )

    fb_url = forms.URLField(
        required=False,
        widget=forms.URLInput(
            attrs={
                "placeholder": "Facebook URL",
                "class": "form-control"
            }
        ))

    twitter_url = forms.URLField(
        required=False,
        widget=forms.URLInput(
            attrs={
                "placeholder": "Twitter URL",
                "class": "form-control"
            }
        ))

    linkedin_url = forms.URLField(
        required=False,
        widget=forms.URLInput(
            attrs={
                "placeholder": "Linkedin URL",
                "class": "form-control"
            }
        ))

    username = forms.CharField(
        widget=forms.TextInput(
            attrs={
                "placeholder": "Username",
                "class": "form-control"
            }
        ))
    email = forms.EmailField(
        widget=forms.EmailInput(
            attrs={
                "placeholder": "Email",
                "class": "form-control"
            }
        ))
    password1 = forms.CharField(
        widget=forms.PasswordInput(
            attrs={
                "placeholder": "Password",
                "class": "form-control"
            }
        ))
    password2 = forms.CharField(
        widget=forms.PasswordInput(
            attrs={
                "placeholder": "Password check",
                "class": "form-control"
            }
        ))

    class Meta:
        model = User
        fields = ('username', 'first_name', 'last_name', 'email', 'password1', 'password2',
                  "avatar", "d_birth", "gender", "fb_url", "twitter_url", "linkedin_url")

    def save(self, commit=True):
        user = super().save(commit=False)
        user_profile = UserProfile(
            user=user, avatar=self.avatar, d_birth=self.cleaned_data['d_birth'], gender=self.cleaned_data['gender'], fb_url=self.cleaned_data['fb_url'], twitter_url=self.cleaned_data['twitter_url'], linkedin_url=self.cleaned_data['linkedin_url'])
        if commit:
            user.save()
            user_profile.save()
        return user


class UpdateUserForm(UserChangeForm):
    username = forms.CharField(
        disabled=True,
        widget=forms.TextInput(
            attrs={
                "placeholder": "Username",
                "class": "form-control"
            }
        ))

    first_name = forms.CharField(
        widget=forms.TextInput(
            attrs={
                "placeholder": "Firstname",
                "class": "form-control",
                "autofocus": "true"
            }
        ))
    last_name = forms.CharField(
        widget=forms.TextInput(
            attrs={
                "placeholder": "Lastname",
                "class": "form-control"
            }
        ))

    email = forms.EmailField(
        widget=forms.EmailInput(
            attrs={
                "placeholder": "Email",
                "class": "form-control"
            }
        ))

    class Meta:
        model = User
        fields = ('username', 'first_name', 'last_name', 'email')


class UpdateUserProfileForm(UserChangeForm):
    d_birth = forms.DateField(
        widget=forms.DateInput(
            attrs={
                'type': 'date',
                "class": "form-control"
            }
        ))

    gender = forms.ChoiceField(
        widget=forms.RadioSelect(attrs={'class': 'custom-control-input'}),
        choices=[('M', 'Male'), ('F', 'Female')]
    )

    fb_url = forms.URLField(
        required=False,
        widget=forms.URLInput(
            attrs={
                "placeholder": "Facebook URL",
                "class": "form-control"
            }
        ))

    twitter_url = forms.URLField(
        required=False,
        widget=forms.URLInput(
            attrs={
                "placeholder": "Twitter URL",
                "class": "form-control"
            }
        ))

    linkedin_url = forms.URLField(
        required=False,
        widget=forms.URLInput(
            attrs={
                "placeholder": "Linkedin URL",
                "class": "form-control"
            }
        ))
    avatar = forms.ImageField(
        required=False,
        widget=forms.FileInput(
            attrs={'class': 'custom-file-input',
                   'style': 'display:none',
                   'id': 'avatar'}
        ))

    class Meta:
        model = UserProfile
        fields = ("avatar", "d_birth", "gender", "fb_url", "twitter_url", "linkedin_url")

    def __init__(self, *args, **kwargs):
        gender = kwargs.pop('gender', None)
        super().__init__(*args, **kwargs)

        if gender:
            self.fields['gender'].widget.attrs['checked'] = gender
    
