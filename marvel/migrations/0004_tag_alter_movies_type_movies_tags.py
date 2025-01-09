# Generated by Django 5.1 on 2024-08-29 16:29

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('marvel', '0003_alter_movies_type'),
    ]

    operations = [
        migrations.CreateModel(
            name='Tag',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=20, unique=True)),
            ],
        ),
        migrations.AlterField(
            model_name='movies',
            name='type',
            field=models.CharField(choices=[('Movie', 'Movie'), ('Series', 'Series'), ('Animatied-movie', 'Animati-movie'), ('Animatied-series', 'Animation-series')], default='Movie', max_length=20),
        ),
        migrations.AddField(
            model_name='movies',
            name='tags',
            field=models.ManyToManyField(blank=True, related_name='movies', to='marvel.tag'),
        ),
    ]
