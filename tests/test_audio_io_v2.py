



class InlineNativeExecutor:
    async def run(self, function, *args, **kwargs):
        return function(*args, **kwargs)
