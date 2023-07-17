from txt2img import txt2img

image = txt2img(
    checkpoint = "juggernaut_final",
    positive = """
        old man
        <lora:animeScreencapStyle_v230epochs:1>
    """,
    negative = """""",
    steps = 10,
    seed = 3125748766
)

image.save("output.png")