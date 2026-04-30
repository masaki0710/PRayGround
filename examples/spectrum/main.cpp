#include "app.h"

int main()
{
    pgSetAppDir(APP_DIR);
    pgSetAppName(APP_NAME_DEFINE);

    auto window = make_shared<Window>("Spectrum", 1080, 720);
    auto app = make_shared<App>();

    pgRunApp(app, window);

    return 0;
}