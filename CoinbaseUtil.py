import os


def get_api_keys():
    crypto_folder = os.getenv("CRYPTO")
    print("crypto_folder")
    print(crypto_folder)
    fd = open(os.path.join(crypto_folder, "jc.txt"))
    lines = []
    for line in fd:
        lines += [line[:-1]]

    api_key = lines[0]
    api_secret = lines[1]
    return (api_key, api_secret)

def get_twelvedata_api_keys():
    crypto_folder = os.getenv("CRYPTO")
    print("crypto_folder")
    print(crypto_folder)
    fd = open(os.path.join(crypto_folder, "jc2.txt"))
    for line in fd:
        api_key = line[:-1]
        break

    return api_key