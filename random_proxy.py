import random

def random_proxy():
    proxies = [
        'http://10.10.1.10:3128',
        'http://10.10.1.11:8080',
        'http://10.10.1.12:8080'
    ]
    return random.choice(proxies)

if __name__ == '__main__':
    print(random_proxy())