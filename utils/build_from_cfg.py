import  inspect
# 接受cfg和registry，执行具体的模块类初始化工作
def build_from_cfg(cfg,registry):
    args = cfg.copy()
    # 获取registry中的模块名
    module_type = args.pop('type')
    
    if isinstance(module_type,str):
        # 如果是字符串，则获取registry中的module
        module_cls = registry.get(module_type)
        if module_cls is None:
            raise KeyError('{} is not registered in registry'.format(module_type))
    elif inspect.isclass(module_type):
        # 如果已经实例化，则直接使用
        module_cls = module_type
        

    else:
        raise TypeError('module_type must be a str or class,but get {}'.format(type(module_type)))

    # 根据cfg实例化module
    return module_cls(**args)
    

