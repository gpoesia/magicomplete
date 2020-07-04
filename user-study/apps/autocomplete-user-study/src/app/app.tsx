import React, { useEffect, useState } from 'react';
import TypingTask from './TypingTask';
import Playground from './Playground';

import './app.scss';

export const App = () => {
    if (location.href.endsWith('/playground.html')) {
        return <Playground />;
    } else {
        return <TypingTask />;
    }
};
export default App;
