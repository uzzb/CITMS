# Generated by Django 5.1.1 on 2024-10-29 00:01

import django.db.models.deletion
from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Location',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=255)),
                ('description', models.TextField(blank=True)),
            ],
            options={
                'db_table': 'locations',
            },
        ),
        migrations.CreateModel(
            name='AnalysisRecord',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('timestamp', models.DateTimeField(auto_now_add=True)),
                ('zone_counts', models.TextField()),
                ('warnings', models.TextField()),
                ('crowd_density', models.FloatField()),
                ('total_count', models.IntegerField()),
                ('velocity', models.FloatField()),
                ('abnormal_events', models.TextField()),
                ('location', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='traffic_management.location')),
            ],
            options={
                'db_table': 'analysis_records',
                'ordering': ['-timestamp'],
            },
        ),
    ]
