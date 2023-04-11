from django.core.management.base import BaseCommand
from apps.home.seed import seed_dataSecteur, seed_dataSociete

class Command(BaseCommand):
    help = 'Seed the database with initial data'

    def handle(self, *args, **options):
        seed_dataSecteur()
        seed_dataSociete()
        
