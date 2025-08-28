# JWT Authentication Configuration
import jwt
import os
from datetime import datetime, timedelta
from typing import Dict, Any, Optional

class JWTAuth:
    """JWT Authentication handler"""
    
    def __init__(self):
        self.secret_key = os.getenv("SECRET_KEY")
        self.algorithm = os.getenv("JWT_ALGORITHM", "HS256")
        self.expiration = int(os.getenv("JWT_EXPIRATION", "3600"))
    
    def generate_token(self, user_data: Dict[str, Any]) -> str:
        """Generate JWT token"""
        payload = {
            "user_id": user_data["id"],
            "username": user_data["username"],
            "role": user_data["role"],
            "exp": datetime.utcnow() + timedelta(seconds=self.expiration),
            "iat": datetime.utcnow(),
            "iss": "quantum-planner"
        }
        
        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
    
    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify JWT token"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return payload
        except jwt.ExpiredSignatureError:
            return None
        except jwt.InvalidTokenError:
            return None
    
    def refresh_token(self, token: str) -> Optional[str]:
        """Refresh JWT token"""
        payload = self.verify_token(token)
        if payload:
            # Generate new token with fresh expiration
            user_data = {
                "id": payload["user_id"],
                "username": payload["username"],
                "role": payload["role"]
            }
            return self.generate_token(user_data)
        return None

# Authentication middleware
def require_auth(role: str = None):
    """Authentication decorator"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Extract token from request headers
            token = extract_token_from_request()
            
            auth = JWTAuth()
            payload = auth.verify_token(token)
            
            if not payload:
                return {"error": "Invalid or expired token"}, 401
            
            if role and payload.get("role") != role:
                return {"error": "Insufficient permissions"}, 403
            
            # Add user info to request context
            kwargs["user"] = payload
            return func(*args, **kwargs)
        return wrapper
    return decorator
