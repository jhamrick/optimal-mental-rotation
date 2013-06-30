def selfparams(f):
    def wrapped_func(obj, x1, x2):
        return f(x1.astype('f8', copy=False),
                 x2.astype('f8', copy=False),
                 *obj.params)
    return wrapped_func


class BaseKernel(type):

    def __new__(cls, name, parents, attrs):

        new_attrs = attrs.copy()

        for attr_name in attrs:

            # skip magic methods
            if attr_name.startswith("__"):
                continue

            # skip non-internal methods
            if not attr_name.startswith("_"):
                continue

            # get the function object
            attr = attrs[attr_name]
            if hasattr(attr, '__call__'):
                func = attr
            elif hasattr(attr, '__func__'):
                func = attr.__func__
            else:
                continue

            # remove _ from beginning of name
            new_attr_name = attr_name[1:]
            new_attr = selfparams(func)

            new_attrs[new_attr_name] = new_attr

        if '__call__' not in attrs and 'K' in new_attrs:
            new_attrs['__call__'] = new_attrs['K']

        # create the class
        obj = super(BaseKernel, cls).__new__(
            cls, name, parents, new_attrs)
        return obj
