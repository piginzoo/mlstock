class I:
    def __init__(self, name, tushare_name, cname, category, ttm=None, normalize=False):
        self.name = name
        self.tushare_name = tushare_name
        self.cname = cname
        self.category = category
        self.ttm = ttm
        self.normalize = normalize
