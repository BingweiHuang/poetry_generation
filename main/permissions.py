# blog/permissions.py
from rest_framework import permissions

class IsAuthorOrReadOnly(permissions.BasePermission):
    """
    自定义权限只允许对象的创建者才能编辑它。"""
    def has_object_permission(self, request, view, obj):

        # 读取权限被允许用于任何请求，
        # 所以我们始终允许 GET，HEAD 或 OPTIONS 请求。
        if request.method in permissions.SAFE_METHODS:
            return True
        # 修改和删除权限只允许给 article 的作者。

        return obj.author_id == request.user.id

class IsFanOrReadOnly(permissions.BasePermission):
    def has_object_permission(self, request, view, obj):

        # 读取权限被允许用于任何请求，
        # 所以我们始终允许 GET，HEAD 或 OPTIONS 请求。
        if request.method in permissions.SAFE_METHODS:
            return True
        # 修改和删除权限只允许给 关注者本人。
        return obj.fan_id == request.user.id

class IsUserOrReadOnly(permissions.BasePermission):

    def has_permission(self, request, view):

        return request.user is not None

    def has_object_permission(self, request, view, obj):

        # 读取权限被允许用于任何请求，
        # 所以我们始终允许 GET，HEAD 或 OPTIONS 请求。
        if request.method in permissions.SAFE_METHODS:
            return True
        # 修改权限只允许给 用户本人。

        return obj.user_id == request.user.id

class DelCommentOrReadOnly(permissions.BasePermission):
    def has_permission(self, request, view):
        return request.user is not None

    def has_object_permission(self, request, view, obj):

        # 读取权限被允许用于任何请求，
        # 所以我们始终允许 GET，HEAD 或 OPTIONS 请求。
        if request.method in permissions.SAFE_METHODS:
            return True
        # 评论删除权限给 帖主和评论者。
        return obj.post.author_id == request.user.id or obj.author_id == request.user.id

class StaffOnly(permissions.BasePermission):
    def has_permission(self, request, view):
        return request.user.is_staff == True


class AdminOnly(permissions.BasePermission):
    def has_permission(self, request, view):
        return request.user.is_superuser == True