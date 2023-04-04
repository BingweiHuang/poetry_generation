"""
Django settings for poetry_generation project.

Generated by 'django-admin startproject' using Django 4.1.

For more information on this file, see
https://docs.djangoproject.com/en/4.1/topics/settings/

For the full list of settings and their values, see
https://docs.djangoproject.com/en/4.1/ref/settings/
"""
from datetime import timedelta
from pathlib import Path

# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent.parent


# Quick-start development settings - unsuitable for production
# See https://docs.djangoproject.com/en/4.1/howto/deployment/checklist/

# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY = 'django-insecure-b5mp&c&zoxu+malks!fe5m-s&y+a662$cg_-%b*1hxr=12m!*p'

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = True

from corsheaders.defaults import default_headers
ALLOWED_HOSTS = ['*'] # 允许所有ip访问
CORS_ALLOW_CREDENTIALS = True
CORS_ORIGIN_ALLOW_ALL = True #所有域名都可以跨域访问
# CORS_ALLOW_HEADERS = ('*') #允许所有的请求头
CORS_ALLOW_HEADERS = default_headers + (
    'Loading',
    'Authorization',
)

# Application definition

INSTALLED_APPS = [
    'corsheaders',
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
]

MIDDLEWARE = [
    'corsheaders.middleware.CorsMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'poetry_generation.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [BASE_DIR / 'templates']
        ,
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

WSGI_APPLICATION = 'poetry_generation.wsgi.application'


# Database
# https://docs.djangoproject.com/en/4.1/ref/settings/#databases

DATABASES = {
    # 'default': {
    #     'ENGINE': 'django.db.backends.sqlite3',
    #     'NAME': BASE_DIR / 'db.sqlite3',
    # }

    'default': {
        # 连接本地mysql数据库
        'ENGINE': 'django.db.backends.mysql',
        'NAME': 'poetry_django',# 你的数据库名
        # 'NAME': 'poetry',# 你的数据库名
        'USER': 'root',# 你的用户名
        'PASSWORD': 'Wei909140058',#你的密码
        'HOST': 'localhost',# 本地连接
        'PORT': '3306',# 本地端口号
        'serverTimezone': 'Asia/Shanghai' # 时区

    }
}


# Password validation
# https://docs.djangoproject.com/en/4.1/ref/settings/#auth-password-validators

AUTH_PASSWORD_VALIDATORS = [
    {
        'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator',
    },
]


# Internationalization
# https://docs.djangoproject.com/en/4.1/topics/i18n/

LANGUAGE_CODE = 'en-us'

TIME_ZONE = 'Asia/Shanghai'

USE_I18N = True

USE_TZ = False


# Static files (CSS, JavaScript, Images)
# https://docs.djangoproject.com/en/4.1/howto/static-files/

STATIC_URL = 'static/'

# Default primary key field type
# https://docs.djangoproject.com/en/4.1/ref/settings/#default-auto-field

DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

REST_FRAMEWORK = {
    # 'DEFAULT_PERMISSION_CLASSES': [
    #     # 'rest_framework.permissions.IsAuthenticated', # 默认所有接口都需要token认证
    #     'rest_framework.permissions.AllowAny',
    # ],

    # 默认的响应渲染类
    'DEFAULT_RENDERER_CLASSES': [
        'rest_framework.renderers.JSONRenderer', # json渲染器，默认返回json数据
        'rest_framework.renderers.BrowsableAPIRenderer',# 浏览器的API渲染器，返回调试界面
    ],

    # 认证
    'DEFAULT_AUTHENTICATION_CLASSES': (
        'rest_framework_simplejwt.authentication.JWTAuthentication', # simplejwt 认证
        'rest_framework.authentication.BasicAuthentication',
        'rest_framework.authentication.SessionAuthentication',
    ),

    # 分页
    'DEFAULT_PAGINATION_CLASS': 'main.pagination.MyLimitOffsetPagination',

    # filter查询
    'DEFAULT_FILTER_BACKENDS': [
        'django_filters.rest_framework.DjangoFilterBackend',
        'rest_framework.filters.SearchFilter',
        'rest_framework.filters.OrderingFilter',
    ],

    # 限流
    'DEFAULT_THROTTLE_CLASSES': [
        'rest_framework.throttling.AnonRateThrottle',
        'rest_framework.throttling.UserRateThrottle',
        'rest_framework.throttling.ScopedRateThrottle',
    ],
    'DEFAULT_THROTTLE_RATES': {
        'anon': '20/min', # 针对未登录用户的全局限流
        'user': '30/min', # 针对已登录用户的全局限流

        'anon_email': '2/min', # 针对未登录用户的发邮件限流
        'user_email': '2/min', # 针对已登录用户的发邮件限流

        'AI_api': '2/min', # 针对AI作诗接口的限流
    },

}

AUTHENTICATION_BACKENDS = (
    'main.MyCustomBackend.MyCustomBackend',
)

SIMPLE_JWT = {
    'ACCESS_TOKEN_LIFETIME': timedelta(minutes=5),
    'REFRESH_TOKEN_LIFETIME': timedelta(days=1),
    'ROTATE_REFRESH_TOKENS': False,
    'BLACKLIST_AFTER_ROTATION': False,
    'UPDATE_LAST_LOGIN': False,

    'ALGORITHM': 'HS256',
    'SIGNING_KEY': 'JdsfjdsfkjSDfhsdf89123SFDf',
    'VERIFYING_KEY': None,
    'AUDIENCE': None,
    'ISSUER': None,
    'JWK_URL': None,
    'LEEWAY': 0,

    'AUTH_HEADER_TYPES': ('Bearer',),
    'AUTH_HEADER_NAME': 'HTTP_AUTHORIZATION',
    'USER_ID_FIELD': 'id',
    'USER_ID_CLAIM': 'user_id',
    # 'USER_ID_CLAIM': 'account_id',
    'USER_AUTHENTICATION_RULE': 'rest_framework_simplejwt.authentication.default_user_authentication_rule',

    'AUTH_TOKEN_CLASSES': ('rest_framework_simplejwt.tokens.AccessToken',),
    'TOKEN_TYPE_CLAIM': 'token_type',
    'TOKEN_USER_CLASS': 'rest_framework_simplejwt.models.TokenUser',

    'JTI_CLAIM': 'jti',

    'SLIDING_TOKEN_REFRESH_EXP_CLAIM': 'refresh_exp',
    'SLIDING_TOKEN_LIFETIME': timedelta(minutes=5),
    'SLIDING_TOKEN_REFRESH_LIFETIME': timedelta(days=1),
}

# 设置邮件域名 发送邮件服务器：smtp.qq.com
EMAIL_HOST = 'smtp.qq.com'
# 设置端口号，为数字  使用SSL，端口号465或587
EMAIL_PORT = 587
# 设置发件人邮箱
EMAIL_HOST_USER = '1073224563@qq.com'
# 设置发件人授权码
EMAIL_HOST_PASSWORD = 'izppavsyltdlbfca'
# EMAIL_HOST_PASSWORD = 'gjgxddwdktmbbajj'
# 设置是否启用安全连接
EMAIL_USE_TLS = False

EMAIL_FROM = '1073224563@qq.com'


# DRF扩展
REST_FRAMEWORK_EXTENSIONS = {
    # 默认缓存时间
    'DEFAULT_CACHE_RESPONSE_TIMEOUT': 60 * 60 * 24,
    # 缓存存储
    'DEFAULT_USE_CACHE': 'default',
}

CACHES = {
    # 'default': {
    #     'BACKEND': 'django.core.cache.backends.db.DatabaseCache',
    #     'LOCATION': 'django_cache',  # 验证码缓存表名
    #     'TIMEOUT': 300,  # 秒 默认情况下缓存键永不过时
    #     'OPTIONS': {
    #         'MAX_ENTRIES': 1000,  # 删除旧值之前允许缓存的最大条目。默认是 300
    #         'CULL_FREQUENCY': 2,  # 缓存条数达到最大值时，删除1/x的缓存数据 max_entries*(1/cull_frequency)
    #     }
    # },

    'default': {
        'BACKEND': 'django_redis.cache.RedisCache',
        # 'LOCATION': 'redis://1.12.62.89:6379', # redis所在服务器或容器ip地址  腾讯云
        'LOCATION': 'redis://124.71.12.157:6379', # redis所在服务器或容器ip地址 华为云
        "OPTIONS": {
            "CLIENT_CLASS": "django_redis.client.DefaultClient",
             "PASSWORD": "Wei909140058", # 你设置的密码
        },
    },
}

REDIS_TIMEOUT=24*60*60
CUBES_REDIS_TIMEOUT=60*30
NEVER_REDIS_TIMEOUT=365*24*60*60
