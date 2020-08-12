import React from "react";

export default class Consent extends React.Component {
    render() {
        return (
            <div className="instructions-container">
                <p>
                    In this experiment, you will type code using different text editor features.
                    Before starting the game task, you will be given further
                    instructions. It should take
                    approximately 20 minutes, including the time it takes to
                    read instructions.
                    Please note that you may only do this task one time.
                </p>
                <p>
                    By following the typing task instructions, you are
                    participating in a study being performed by cognitive scientists
                    in the Stanford Department of Psychology. If you have questions
                    about this research, please contact Gabriel Poesia
                    at poesia@stanford.edu
                    or Noah Goodman, at ngoodman@stanford.edu. You must be at least
                    18 years old to participate. Your participation in this research
                    is voluntary. You may decline to answer any or all of the
                    following questions. You may decline further participation, at
                    any time, without adverse consequences. Your anonymity is
                    assured; the researchers who have requested your participation
                    will not receive any personal information about you.
                </p>
                <p>
                    <input type="checkbox" />
                    I have read and agree with the above.
                </p>
                <p>
                    <button onClick={() => this.props.start()}>Continue</button>
                </p>
            </div>
        )
    }
}