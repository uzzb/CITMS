# Generated by Django 5.1.1 on 2024-10-29 04:11

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('traffic_management', '0001_initial'),
    ]

    operations = [
        migrations.AddField(
            model_name='analysisrecord',
            name='acceleration',
            field=models.FloatField(default=0.0),
        ),
        migrations.AddField(
            model_name='analysisrecord',
            name='max_zone_density',
            field=models.FloatField(default=0.0),
        ),
        migrations.AddField(
            model_name='analysisrecord',
            name='peak_hours',
            field=models.TextField(default='[]'),
        ),
    ]