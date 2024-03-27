from tinygrad import Tensor, nn


class UNet:
    def __init__(self, out_classes: int = 1, base_filters: int = 8):
        self.base_filters = base_filters

        self.conv_b = self.conv_block(3, self.base_filters)
        self.conv_2b = self.conv_block(self.base_filters, 2 * self.base_filters)
        self.conv_4b = self.conv_block(2 * self.base_filters, 4 * self.base_filters)
        self.conv_8b = self.conv_block(4 * self.base_filters, 8 * self.base_filters)
        self.conv_16b = self.conv_block(8 * self.base_filters, 16 * self.base_filters)
        self.conv_b_ = self.conv_block(2 * self.base_filters, self.base_filters)
        self.conv_2b_ = self.conv_block(4 * self.base_filters, 2 * self.base_filters)
        self.conv_4b_ = self.conv_block(8 * self.base_filters, 4 * self.base_filters)
        self.conv_8b_ = self.conv_block(16 * self.base_filters, 8 * self.base_filters)

        self.max_pool = lambda x: x.max_pool2d(kernel_size=2, stride=2)

        self.conv_t_1 = nn.ConvTranspose2d(
            16 * self.base_filters, 8 * self.base_filters, 2, 2
        )
        self.conv_t_2 = nn.ConvTranspose2d(
            8 * self.base_filters, 4 * self.base_filters, 2, 2
        )
        self.conv_t_3 = nn.ConvTranspose2d(
            4 * self.base_filters, 2 * self.base_filters, 2, 2
        )
        self.conv_t_4 = nn.ConvTranspose2d(
            2 * self.base_filters, self.base_filters, 2, 2
        )

        self.out = nn.Conv2d(self.base_filters, out_classes, 1)

    def conv_block(self, inp, out):
        return [
            nn.Conv2d(inp, out, 3, padding=1),
            lambda x: x.relu(),
            nn.Conv2d(out, out, 3, padding=1),
            lambda x: x.relu(),
        ]

    def __call__(self, x: Tensor) -> Tensor:

        conv_1 = x.sequential(self.conv_b)
        pool_1 = self.max_pool(conv_1)

        conv_2 = pool_1.sequential(self.conv_2b)
        pool_2 = self.max_pool(conv_2)

        conv_3 = pool_2.sequential(self.conv_4b)
        pool_3 = self.max_pool(conv_3)

        conv_4 = pool_3.sequential(self.conv_8b)
        pool_4 = self.max_pool(conv_4)

        conv_5 = pool_4.sequential(self.conv_16b)

        up_6 = self.conv_t_1(conv_5)
        merge_6 = conv_4.cat(up_6, dim=1)
        conv_6 = merge_6.sequential(self.conv_8b_)

        up_7 = self.conv_t_2(conv_6)
        merge_7 = conv_3.cat(up_7, dim=1)
        conv_7 = merge_7.sequential(self.conv_4b_)

        up_8 = self.conv_t_3(conv_7)
        merge_8 = conv_2.cat(up_8, dim=1)
        conv_8 = merge_8.sequential(self.conv_2b_)

        up_9 = self.conv_t_4(conv_8)
        merge_9 = conv_1.cat(up_9, dim=1)
        conv_9 = merge_9.sequential(self.conv_b_)

        out = self.out(conv_9).sigmoid()

        return out


model = UNet()
