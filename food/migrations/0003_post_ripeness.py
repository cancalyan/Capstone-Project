# Generated by Django 4.1.5 on 2023-06-08 20:23

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('food', '0002_remove_post_content_remove_post_title_post_image_and_more'),
    ]

    operations = [
        migrations.AddField(
            model_name='post',
            name='ripeness',
            field=models.DecimalField(decimal_places=2, default=0.0, max_digits=3),
        ),
    ]