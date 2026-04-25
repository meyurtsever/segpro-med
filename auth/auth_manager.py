"""
Authentication utilities for the crowdsourcing system

Passwords are stored as bcrypt hashes.  Legacy plaintext passwords are
transparently migrated to bcrypt on first successful login.
"""

import os
import csv
import logging
import bcrypt
import tempfile

logger = logging.getLogger(__name__)


def _is_bcrypt_hash(value: str) -> bool:
    """Return True if *value* looks like a bcrypt hash."""
    return value.startswith("$2b$") or value.startswith("$2a$")


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
    
    def _verify_password(self, plain_password: str, stored_password: str) -> bool:
        """Verify a password against the stored value (bcrypt hash or legacy plaintext)."""
        if _is_bcrypt_hash(stored_password):
            return bcrypt.checkpw(
                plain_password.encode('utf-8'),
                stored_password.encode('utf-8'),
            )
        # Legacy plaintext comparison (will be migrated on success)
        return stored_password == plain_password
    
    def _hash_password(self, plain_password: str) -> str:
        """Return bcrypt hash for *plain_password*."""
        return bcrypt.hashpw(
            plain_password.encode('utf-8'),
            bcrypt.gensalt(),
        ).decode('utf-8')
    
    def _migrate_password(self, username: str, plain_password: str):
        """Replace a legacy plaintext password with its bcrypt hash on disk.
        
        Uses atomic write (write-to-temp then rename) so a crash cannot
        corrupt the user file.
        """
        new_hash = self._hash_password(plain_password)
        self.users[username]['password'] = new_hash
        
        try:
            dir_name = os.path.dirname(os.path.abspath(self.users_file_path))
            fd, tmp_path = tempfile.mkstemp(dir=dir_name, suffix='.tmp')
            with os.fdopen(fd, 'w', newline='', encoding='utf-8') as tmp_file:
                writer = csv.DictWriter(tmp_file, fieldnames=['id', 'password', 'role', 'expert_score'])
                writer.writeheader()
                for uid, udata in self.users.items():
                    writer.writerow({
                        'id': uid,
                        'password': udata['password'],
                        'role': udata['role'],
                        'expert_score': udata['expert_score'],
                    })
            # Atomic replace (Windows: os.replace is atomic on NTFS)
            os.replace(tmp_path, self.users_file_path)
            logger.info(f"Migrated password for user '{username}' to bcrypt hash")
        except Exception as e:
            logger.error(f"Failed to migrate password for '{username}': {e}")
            # Ensure temp file is cleaned up on failure
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

    def authenticate(self, username, password):
        """Authenticate user credentials"""
        if username in self.users:
            stored = self.users[username]['password']
            if self._verify_password(password, stored):
                # Migrate legacy plaintext password to bcrypt on first success
                if not _is_bcrypt_hash(stored):
                    self._migrate_password(username, password)
                return {
                    'user_id': username,
                    'role': self.users[username]['role'],
                    'expert_score': self.users[username]['expert_score']
                }
        return None
    
    def get_experts(self):
        """Get list of all expert users"""
        return [
            user_id for user_id, user_data in self.users.items()
            if user_data['role'] == 'expert'
        ]
