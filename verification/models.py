# verification/models.py
from django.db import models

class Signature(models.Model):
    signature_image = models.ImageField(upload_to='signatures/')
    uploaded_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Signature {self.id} uploaded at {self.uploaded_at}"
