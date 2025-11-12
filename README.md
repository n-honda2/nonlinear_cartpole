# nonlinear_cartpole
Graduation project applying traditional linear and non linear control methods using gymnasium cartpole

## Ambiente virtual de Python

Para rodar o projeto em um ambiente virtual com o módulo [venv](https://docs.python.org/pt-br/3/library/venv.html#module-venv), primeiramente instale-o da seguinte forma:

```bash
sudo apt install python3-venv
```

Então para criar o abiente virtual faça:

```bash
python3 -m venv venv
```

Então para ativar o ambiente virtual pelo `bash`, faça:

```bash
source ./venv/bin/activate
```

Já pelo fish, faça:

```bash
source ./venv/bin/activate.fish
```

Por fim, caso o arquivo `requirements.txt` não exista ou caso seja necessário atualizá-lo, com todas as depências necessárias instaladas manualmente no ambiente virtual, gere o arquivo fazendo:

```bash
pip3 freeze -l > requirements.txt
```

Caso o arquivo  já exista, para instalar as dependências, rode o seguinte comando:

```bash
pip3 install -r requirements.txt
```

Para desativar o ambiente virtual, rode no terminal:

```bash
deactivate
```

## Rodando o pacote

Para rodar o pacote, estando na raiz do repositório, faça o seguinte:

```bash
python3 main.py
```
