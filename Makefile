PROJECT = lenet_m4
BUILD_DIR = build

CFILES = target_m4/lenet.c

# Edit these two lines as needed
DEVICE=stm32l496rgt3
OOCD_FILE = board/stm32l4-generic.cfg

# All lines below probably should not be edited
VPATH += $(TA_EXPT_DIR)
INCLUDES += $(patsubst %,-I%, . $(TA_EXPT_DIR))
OPENCM3_DIR=libopencm3

include $(OPENCM3_DIR)/mk/genlink-config.mk
include rules.mk
include $(OPENCM3_DIR)/mk/genlink-rules.mk
