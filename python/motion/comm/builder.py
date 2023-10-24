from motion.utils.registry import Registry


SERVERS = Registry("server")


def build_server(server_type, *args, **kwargs):
    server = SERVERS.get(server_type)(*args, **kwargs)
    return server
