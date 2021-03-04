package players.portfolio.scripts;

import core.actions.Action;
import core.actors.Actor;
import core.game.GameState;

public class AttackMaxDamageScr extends BaseScript {

    //This script returns the attack action that would cause more damage. Bonus if
    //  that damage also kills the enemy unit.

    @Override
    public Action process(GameState gs, Actor ac) {
        return new Func().getActionByActorAttr(gs, actions, ac, Feature.DAMAGE, true);
    }

}
