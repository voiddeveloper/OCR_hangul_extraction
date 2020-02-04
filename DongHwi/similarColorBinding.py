def similarColorBinding(image, devide):
    height, width, channel = image.shape

    for i in range(height):
        for j in range(width):
            b, g, r = image[i][j]
            # print(j, ': ',b, g, r)

            value = int(255 / devide)
            b = (value * int((b / value)))
            g = (value * int((g / value)))
            r = (value * int((r / value)))

            if b >= 255:
                b = 255
            else:
                b = b

            if g >= 255:
                g = 255
            else:
                g = g

            if r >= 255:
                r = 255
            else:
                r = r

            image[i][j] = (b, g, r)

    return image
