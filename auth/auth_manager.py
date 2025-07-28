"""
Authentication utilities for the crowdsourcing system
"""

import os
import csv
import logging

logger = logging.getLogger(__name__)

class AuthManager:
    """Manages user authentication and session state"""
    
    def __init__(self, users_file_path="db/users.txt"):
        self.users_file_path = users_file_path
        self.users = self._load_users()
    
    def _load_users(self):
        """Load users from the users.txt file"""
        users = {}
        if not os.path.exists(self.users_file_path):
            logger.warning(f"Users file not found: {self.users_file_path}")
            return users
        
        try:
            with open(self.users_file_path, 'r', newline='', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    users[row['id']] = {
                        'password': row['password'],
                        'role': row['role'],
                        'expert_score': float(row['expert_score'])
                    }
            logger.info(f"Loaded {len(users)} users from {self.users_file_path}")
        except Exception as e:
            logger.error(f"Error loading users: {e}")
        
        return users
    
    def authenticate(self, username, password):
        """Authenticate user credentials"""
        if username in self.users:
            if self.users[username]['password'] == password:
                return {
                    'user_id': username,
                    'role': self.users[username]['role'],
                    'expert_score': self.users[username]['expert_score']
                }
        return None
    
    def get_experts(self):
        """Get list of all expert users"""
        return {
            user_id: user_data for user_id, user_data in self.users.items()
            if user_data['role'] == 'expert'
        }
