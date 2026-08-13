import franky

# 1. 查看枚举里有哪些成员
print(dir(franky.ReferenceType))
# 输出中会包含 'Absolute' 和 'Relative'

# 2. 查看详细的 docstring 说明（如果绑定的 C++ 源码有写注释的话）
help(franky.ReferenceType)