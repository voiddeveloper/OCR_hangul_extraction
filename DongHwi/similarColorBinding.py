# 저해상도 이미지로 바꾸기
# 256 * 256 * 256 개수의 색을 - > devide * devide * devide 개수의 색깔로 바꾼다.
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

# 사용법
# ex) bind_image = similarColorBinding( bgr 이미지, 4 ) => 이미지를 구성하는 색의 최대 갯수 : 4*4*4 개
# ex) bind_image = similarColorBinding( bgr 이미지, 6 ) => 이미지를 구성하는 색의 최대 갯수 : 6*6*6 개
