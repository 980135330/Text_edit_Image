import sys
sys.path.append("..")
from utils import Registry,build_from_cfg

OPTIMIZER = Registry('optimizer')

def build_optimizer(params,cfg,registry=OPTIMIZER):

    args = cfg.copy()
    # 获取registry中的模块名
    module_type = args.pop('type')
    
    if isinstance(module_type,str):
        # 如果是字符串，则获取registry中的module
        module_cls = registry.get(module_type)
        if module_cls is None:
            raise KeyError('{} is not registered in registry'.format(module_type))
    else:
        raise TypeError('module_type must be a str or class,but get {}'.format(type(module_type)))

    # 根据cfg实例化module
    return module_cls(params,**args)

    