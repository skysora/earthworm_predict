import requests

def test(token):
    message = 'test: \n'

    while True:
        try:
            url = "https://notify-api.line.me/api/notify"
            headers = {
                'Authorization': f'Bearer {token}'
            }
            payload = {
                'message': message,
            }
            response = requests.request(
                "POST",
                url,
                headers=headers,
                data=payload,
            )
            if response.status_code == 200:
                print(f"Success, system is alive -> {response.text}")
                break
            else:
                print(f'(Alive) Error -> {response.status_code}, {response.text}')
        except Exception as e:
            print(e)
    return 

if __name__ == '__main__':
    while True:
        token = input('token: (0: end)')

        if token == '0':
            break

        test(token)