# Generated by Django 3.2.16 on 2023-04-08 15:57

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('home', '0006_remove_historical_data_rendement'),
    ]

    operations = [
        migrations.AddField(
            model_name='historical_data',
            name='Rendement',
            field=models.FloatField(default=0),
        ),
        migrations.AddField(
            model_name='historical_data',
            name='Rendement_moyen',
            field=models.FloatField(default=0),
        ),
    ]
