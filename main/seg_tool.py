import pygame
from pygame.locals import *
from sys import exit
from common.common import *
import utils.screen_draw as screen_draw
import main.event as e


def main():
    pygame.init()
    screen = pygame.display.set_mode(SCREEN_SIZE, 0, 32)
    pygame.display.set_caption(SCREEN_TITLE)
    screen_draw.init_screen(screen)
    current_img = e.button1(screen)
    moving = False
    operation = OPERATION_MODIFY
    while True:
        for event in pygame.event.get():
            if event.type == QUIT:
                # MODEL.save()
                exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                if BUTTON1[0][0] <= event.pos[0] <= BUTTON1[0][1] and BUTTON1[1][0] <= event.pos[1] <= BUTTON1[1][1]:
                    current_img = e.button1(screen)
                    operation = OPERATION_MODIFY
                if BUTTON2[0][0] <= event.pos[0] <= BUTTON2[0][1] and BUTTON2[1][0] <= event.pos[1] <= BUTTON2[1][1]:
                    current_img = e.button2(screen, current_img)
                    operation = OPERATION_MODIFY
                if BUTTON3[0][0] <= event.pos[0] <= BUTTON3[0][1] and BUTTON3[1][0] <= event.pos[1] <= BUTTON3[1][1]:
                    e.button3(screen, current_img)
                    operation = OPERATION_PRE
                if BUTTON4[0][0] <= event.pos[0] <= BUTTON4[0][1] and BUTTON4[1][0] <= \
                        event.pos[1] <= BUTTON4[1][1]:
                    e.button4(screen, current_img)
                    operation = OPERATION_MODIFY
                if (operation == OPERATION_VIEW or operation == OPERATION_MODIFY) and BUTTON5[0][0] <= event.pos[0] <= \
                        BUTTON5[0][1] and BUTTON5[1][0] <= \
                        event.pos[1] <= BUTTON5[1][1]:
                    e.button5(screen)

            if event.type == pygame.MOUSEBUTTONDOWN:
                if operation == OPERATION_PRE:
                    e.get_initial_interactive_point(screen, pygame.mouse.get_pos())
                else:
                    if e.click(pygame.mouse.get_pos()):
                        e.get_interactive_point(screen, current_img, pygame.mouse.get_pos())
                    else:
                        moving = True
            if event.type == pygame.MOUSEBUTTONUP:
                e.get_end_point(screen, current_img, pygame.mouse.get_pos())
                moving = False
            if moving:
                e.modify(screen, pygame.mouse.get_pos(), current_img)

        pygame.display.update()


if __name__ == "__main__":
    main()
