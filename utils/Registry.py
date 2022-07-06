import inspect

class Registry:

    def __init__(self,name):
        # 实现注册器的细分，同类别共享一个类
        self.name = name 
        # 存放具体该类别下具体的module
        self.module_dict = {}
    
    # 执行具体的方法注册
    def _register_module(self,module_class = None,name = None):

        # 传入module必须是class
        if not inspect.isclass(module_class):
            raise TypeError('module_class must be a class,but get {}'.format(type(module_class)))

        # 如果没有指定module_name，则使用module_class的名字
        if name == None:
            name = module_class.__name__
        
        # 如果module_name已经存在，则报错
        if name in self.module_dict:
            raise KeyError('module_name {} already exist'.format(name))

        self.module_dict[name] = module_class
    
    # 装饰器函数，用于注册module
    # 这里我们用register作为装饰函数，接收cls参数，并最终返回cls，所以cls经过装饰没有改变
    # 我们只是给cls增加了一个注册功能

    def register_module(self):
        
        def register(cls,name = None):
            self._register_module(cls,name)
            return cls

        return register
    
    def get(self,module_name):
        return self.module_dict.get(module_name,None)  # 返回module_name对应的module
    
    
        

    

