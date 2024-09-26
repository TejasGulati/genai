from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.exceptions import AuthenticationFailed
from rest_framework import status
from rest_framework_simplejwt.tokens import RefreshToken, AccessToken
from django.conf import settings
from django.shortcuts import get_object_or_404
from users.serializers import UserSerializer
from users.models import User, BlacklistedToken
from rest_framework_simplejwt.exceptions import TokenError
import logging
from rest_framework.permissions import AllowAny, IsAuthenticated
from django.contrib.auth.password_validation import validate_password
from django.core.exceptions import ValidationError
import jwt

# Set up logger
logger = logging.getLogger(__name__)

class RegisterView(APIView):
    permission_classes = [AllowAny]

    def post(self, request):
        serializer = UserSerializer(data=request.data)
        if serializer.is_valid():
            try:
                # Check if email already exists
                if User.objects.filter(email=serializer.validated_data['email']).exists():
                    return Response({'email': ['Email already exists.']}, status=status.HTTP_400_BAD_REQUEST)
                
                # Validate password
                validate_password(serializer.validated_data['password'])
                
                serializer.save()
                return Response(serializer.data, status=status.HTTP_201_CREATED)
            except ValidationError as e:
                return Response({'password': e.messages}, status=status.HTTP_400_BAD_REQUEST)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

class LoginView(APIView):
    permission_classes = [AllowAny]

    def post(self, request):
        email = request.data.get('email')
        password = request.data.get('password')

        if not email:
            return Response({'email': ['Email is required.']}, status=status.HTTP_400_BAD_REQUEST)
        if not password:
            return Response({'password': ['Password is required.']}, status=status.HTTP_400_BAD_REQUEST)

        user = User.objects.filter(email=email).first()

        if user is None:
            return Response({'email': ['User not found.']}, status=status.HTTP_404_NOT_FOUND)

        if not user.check_password(password):
            return Response({'password': ['Incorrect password.']}, status=status.HTTP_400_BAD_REQUEST)

        refresh = RefreshToken.for_user(user)
        access_token = str(refresh.access_token)
        refresh_token = str(refresh)

        response = Response({
            'access': access_token,
            'refresh': refresh_token
        })
        response.set_cookie(key='jwt', value=refresh_token, httponly=True, secure=True)
        return response

class RefreshTokenView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request):
        refresh_token = request.COOKIES.get('jwt')

        if not refresh_token:
            raise AuthenticationFailed("Refresh token is missing!")

        try:
            token = RefreshToken(refresh_token)
            
            # Check if the refresh token is blacklisted
            if BlacklistedToken.objects.filter(token=refresh_token).exists():
                raise AuthenticationFailed("Refresh token is blacklisted!")

            access_token = str(token.access_token)
        except TokenError:
            raise AuthenticationFailed("Invalid refresh token!")

        return Response({'access': access_token})

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.exceptions import AuthenticationFailed
from rest_framework import status
from rest_framework.permissions import IsAuthenticated
from users.serializers import UserSerializer
from users.models import User

class UserView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        user = request.user
        serializer = UserSerializer(user)
        return Response(serializer.data)

    def patch(self, request):
        user = request.user
        serializer = UserSerializer(user, data=request.data, partial=True)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    
class LogoutView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request):
        try:
            refresh_token = request.COOKIES.get('jwt')
            access_token = request.auth.token if hasattr(request, 'auth') and hasattr(request.auth, 'token') else None

            if refresh_token:
                try:
                    token = RefreshToken(refresh_token)
                    BlacklistedToken.objects.create(token=str(token), user=request.user)
                except TokenError:
                    pass  # Token was invalid, continue with the logout process

            if access_token:
                BlacklistedToken.objects.create(token=str(access_token), user=request.user)

            response = Response({"detail": "Successfully logged out."}, status=status.HTTP_200_OK)
            response.delete_cookie('jwt')
            return response

        except Exception as e:
            logger.error(f"Error during logout: {str(e)}")
            return Response({"error": "An error occurred during logout"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)