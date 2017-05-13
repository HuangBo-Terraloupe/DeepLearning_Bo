import datetime
now = datetime.datetime.now


def log(tags, message, *args):
    tag = ":".join(tags)
    message = str(message)
    message = "{}[{}]:{}".format(now(), tag, message)
    message = message % args
    print message

def error(message="", *args):
    log(["ERROR"], message, *args)

def info(message="", *args):
    log(["INFO"], message, *args)

def warn(message="", *args):
    log(["WARNING"], message, *args)

def debug(message, *args):
    log(["DEBUG"], message, *args)


class ProgressLogger:

    def __init__(self, total):
        self.total = total
        self.current = 0
        self.start_time = now()

    def update(self, item="file"):
        self.current += 1
        output = "Processed %s %d/%d" % (item, self.current, self.total)
        if self.current < self.total:

            dt = now() - self.start_time
            remaining = self.total - self.current

            eta = remaining * (dt.seconds / self.current)
            output += " ETA: %d seconds" % eta
            info(output)
        else:
            dt = now() - self.start_time
            output += " Done"
            info(output)
            info("Total time: %s" % dt)



